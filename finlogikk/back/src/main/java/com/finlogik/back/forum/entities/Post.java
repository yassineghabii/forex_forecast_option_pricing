package com.finlogik.back.forum.entities;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.*;

import jakarta.persistence.*;
import java.io.Serializable;
import java.util.Date;
import java.util.Set;

@Entity
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class Post implements Serializable{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idPost;
    private String content;
    private Integer likes;
    private Integer dislikes;

    @Temporal(TemporalType.TIMESTAMP)
    private Date creationDate;
    private Boolean modified;

    private String userId;
    @ManyToOne
    @JsonIgnore
    @JoinColumn(name = "topic_id")
    private Topic topic;
    @OneToMany(mappedBy = "post", cascade = CascadeType.ALL)
    @JsonIgnore
    private Set<Comment> comments;
}
