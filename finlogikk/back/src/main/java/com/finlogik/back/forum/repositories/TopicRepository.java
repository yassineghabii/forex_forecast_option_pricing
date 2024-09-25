package com.finlogik.back.forum.repositories;

import com.finlogik.back.forum.entities.Topic;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface TopicRepository extends CrudRepository<Topic, Long>  {

    @Query("SELECT COUNT(p) FROM Post p WHERE p.topic.idTopic = :idTopic")
    Long countPostsByIdTopic(@Param("idTopic") Long idTopic);

    @Query("SELECT w.idTopic FROM Topic w WHERE w.userId = :userId")
    Long findTopicIdByUserId(@Param("userId") String userId);

}
