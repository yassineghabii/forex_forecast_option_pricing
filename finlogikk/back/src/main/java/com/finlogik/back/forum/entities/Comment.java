package com.finlogik.back.forum.entities;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.*;

import jakarta.persistence.*;
import java.io.Serializable;
import java.util.Date;

@Entity
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class Comment implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idComment;
    private String content;
    private Integer likes;
    private Integer dislikes;
    @Temporal(TemporalType.TIMESTAMP)
    private Date creationDate;
    private Boolean modified;

    private String userId;
    @ManyToOne
    @JsonIgnore
    @JoinColumn(name = "post_id")
    private Post post;

}
